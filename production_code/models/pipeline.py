import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, SimpleRNN
from sklearn.metrics import mean_squared_error

def load_data():
    # Load the dataset using pandas
    data = pd.read_csv('msft.csv')
    return data

def preprocess_data(data):
    # Split the dataset into training and testing sets
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Scale the data using Min-Max normalization
    train_data = (train_data - np.min(train_data)) / (np.max(train_data) - np.min(train_data))
    test_data = (test_data - np.min(test_data)) / (np.max(test_data) - np.min(test_data))

    return train_data, test_data

def train_linear_regression(train_data):
    # Train a linear regression model on the training data
    X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    lr_model = LinearRegression().fit(X_train, y_train)
    return lr_model

def train_lstm(train_data):
    # Train a LSTM model on the training data
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(50, return_sequences=True))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(50))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mse', optimizer='adam')
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)
    return lstm_model

def train_rnn(train_data):
    # Train a RNN model on the training data
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    rnn_model = Sequential()
    rnn_model.add(SimpleRNN(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    rnn_model.add(Dropout(0.2))
    rnn_model.add(SimpleRNN(50, return_sequences=True))
    rnn_model.add(Dropout(0.2))
    rnn_model.add(SimpleRNN(50))
    rnn_model.add(Dropout(0.2))
    rnn_model.add(Dense(1))
    rnn_model.compile(loss='mse', optimizer='adam')
    rnn_model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)
    return rnn_model

def train_gradient_boosting(train_data):
    # Train a gradient boosting regressor model on the training data
    X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    gbr_model = GradientBoost
