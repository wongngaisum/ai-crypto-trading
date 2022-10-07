from backtesting import Backtest, Strategy
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from model_matrix import get_clean_Xy, get_y, get_X
import numpy as np
from sklearn import preprocessing
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
import argparse

np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)
keras.utils.set_random_seed(1234)

parser = argparse.ArgumentParser("backtest")
parser.add_argument("lag", help="lag window of social media indicators", type=int)
parser.add_argument("fee", help="trading fee rate", type=float)
parser.add_argument("close_first", help="do not open opposite position immediately, close first", type=int)
args = parser.parse_args()  

N_TRAIN = 5000
WINDOW = 6

def create_dataset(X, y, time_step=1):
    dataX, dataY = [], []
    for i in range(len(X)-time_step-1):
        a = X[i:(i+time_step)]
        dataX.append(a)
        if y is not None:
            dataY.append(y[i+time_step])
    return np.array(dataX), np.array(dataY)

# Build the RNN model
def build_model(features_count):
    model=keras.models.Sequential() 
    model.add(keras.layers.LSTM(50,return_sequences=True,input_shape=(WINDOW,features_count))) 
    model.add(keras.layers.LSTM(50,return_sequences=True)) 
    # model.add(keras.layers.LSTM(50,return_sequences=True)) 
    model.add(keras.layers.LSTM(50)) 
    model.add(keras.layers.Dense(3, activation='softmax')) 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

class LSTMStrategy(Strategy):
    price_delta = .01
    buy_size = .2
    sell_size = .2
    
    def init(self):       
        # Train the classifier in advance on the first N_TRAIN examples
        df = self.data.df.iloc[:N_TRAIN]
        X, y = get_clean_Xy(df)
                
        # Normalzie
        self.scaler = preprocessing.StandardScaler()
        scaled_X = self.scaler.fit_transform(X)
        
        self.pca = PCA(n_components=0.99, svd_solver='full')
        self.pca = self.pca.fit(scaled_X)
        scaled_X = self.pca.transform(scaled_X)
        
        # self.tree = ExtraTreesClassifier(n_estimators=20, random_state=0)
        # self.tree = self.tree.fit(scaled_X, y)
        # self.tree = SelectFromModel(self.tree, prefit=True)
        # scaled_X = self.tree.transform(scaled_X)
        features_count = scaled_X.shape[1]

        train = create_dataset(scaled_X, y, WINDOW)
        scaled_X = train[0]
        y = train[1]
        
        y = to_categorical(y.astype(int), 3)

        self.clf = build_model(features_count)
        self.clf.fit(
            scaled_X, y, batch_size=64, epochs=20, verbose=2
        )

        # df = self.data.df.iloc[N_TRAIN:]
        # X_test, y_test = get_clean_Xy(df)
        # scaled_X_test = self.scaler.fit_transform(X_test)
        # # scaled_X_test = self.tree.transform(scaled_X_test)
        # scaled_X_test, y_test = create_dataset(scaled_X_test, y_test, WINDOW)
        
        # y_test_pred = self.clf.predict(scaled_X_test)
        # y_test_pred = np.argmax(y_test_pred, axis=1)

        # y_test_pred[y_test_pred == 2] = -1
        
        # print(accuracy_score(y_test, y_test_pred))
        # print(confusion_matrix(y_test, y_test_pred))
        # print(classification_report(y_test, y_test_pred))
        
        # X_test = X_test.iloc[WINDOW+1:]
        # X_test['pred'] = y_test_pred
        # self.y_test_pred = X_test
    
        # Plot y for inspection
        self.y_true = self.I(get_y, self.data.df, name='y_true')

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

            # Update the plotted "forecast" indicator
            X = get_X(self.data.df.iloc[-WINDOW-2:])

            scaled_X = self.scaler.transform(X)
            scaled_X = self.pca.transform(scaled_X)
            
            scaled_X, _ = create_dataset(scaled_X, None, WINDOW)
            
            forecast = self.clf.predict(scaled_X, batch_size=1, verbose=0)
            forecast = np.argmax(forecast[-1])
            print(forecast)
            if forecast == 2:
                forecast = -1

            self.forecasts[-1] = forecast

            self.t.append(int(self.y_true[-1]))
            self.p.append(int(forecast))
            print(self.data.df.shape[0], self.data_len)
            if (self.data.df.shape[0] == self.data_len - 1):
                print(accuracy_score(self.t, self.p))
                print(confusion_matrix(self.t, self.p))
                print(classification_report(self.t, self.p))

            # If our forecast is upwards and we don't already hold a long position
            # place a long order for 20% of available account equity. Vice versa for short.
            # Also set target take-profit and stop-loss prices to be one price_delta
            # away from the current closing price.

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
            
class LSTMWalkForwardStrategy(LSTMStrategy):
    def next(self):
        # Skip the cold start period with too few values available
        if len(self.data) < N_TRAIN:
            return

        # Re-train the model only every 20 iterations.
        # Since 20 << N_TRAIN, we don't lose much in terms of
        # "recent training examples", but the speed-up is significant!
        if len(self.data) % 50:
            return super().next()

        # Retrain on last N_TRAIN values
        df = self.data.df[-N_TRAIN:-1]
        X, y = get_clean_Xy(df)

        # Normalzie
        self.scaler = preprocessing.StandardScaler()
        scaled_X = self.scaler.fit_transform(X)

        self.pca = self.pca.fit(scaled_X)
        scaled_X = self.pca.transform(scaled_X)

        features_count = scaled_X.shape[1]

        train = create_dataset(scaled_X, y, WINDOW)
        scaled_X = train[0]
        y = train[1]
        
        y = to_categorical(y.astype(int), 3)

        self.clf = build_model(features_count)
        self.clf.fit(
            scaled_X, y, batch_size=256, epochs=20, verbose=2
        )


        # Now that the model is fitted, 
        # proceed the same as in MLTrainOnceStrategy
        super().next()