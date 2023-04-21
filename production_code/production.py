from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import yfinance as yf
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import logging


def predict_stock_prices(data, test_size=0.2, random_state=42):
    """
    Predicts stock prices using linear regression.

    Parameters:
        data (pandas.DataFrame): A DataFrame containing the stock price data.
        test_size (float): The proportion of the data to use for testing. Defaults to 0.2.
        random_state (int): The random seed to use for splitting the data. Defaults to 42.

    Returns:
        A tuple containing the predicted stock prices, the regression coefficient,
        the regression intercept, and the regression confidence.
    """
    # Split the data into training and testing sets
    x = data.index
    y = data['Close']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    # Perform linear regression
    lr = LinearRegression()
    lr.fit(x_train.to_numpy().reshape(-1, 1), y_train)
    coef = lr.coef_
    intercept = lr.intercept_
    confidence = lr.score(x_test.to_numpy().reshape(-1,1), y_test)

    # Make predictions using the test set
    predicted = lr.predict(x_test.to_numpy().reshape(-1,1)).reshape(-1)

    # Plot the results
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.scatter(x, y, label='True Values')
    ax.plot(x_test, predicted, label='Predictions', color='orange')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Linear Regression: Predictions vs True Values')
    ax.legend()

    plt.show()

    return predicted, coef, intercept, confidence


""""
predicted, coef, intercept, confidence = predict_stock_prices(msft)
"""

"""
LSTM
"""

class LSTMPredictor:

"""
    Predicts stock prices using a LSTM neural network.

    Parameters:
        data (pandas.DataFrame): A DataFrame containing the stock price data.
        test_size (float): The proportion of the data to use for testing. Defaults to 0.2.
        random_state (int): The random seed to use for splitting the data. Defaults to 42.
        epochs (int): The number of times to iterate over the training dataset. Defaults to 50.
        batch_size (int): The number of samples to use in each batch. Defaults to 32.
        num_units (int): The number of units in each LSTM layer. Defaults to 50.
        activation (str): The activation function to use in the LSTM layers. Defaults to "tanh".
        dropout (float): The proportion of input units to drop during training. Defaults to 0.2.

    Returns:
        A tuple containing the predicted stock prices and the training history.

"""
    def __init__(self, train_data, test_data, lookback=60, lstm_units=100, dense_units=25, epochs=3, batch_size=1):
        self.train_data = train_data
        self.test_data = test_data
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.predictions = None

    def prepare_data(self):
        train_scaled = self.scaler.fit_transform(self.train_data.reshape(-1, 1))
        x_train, y_train = [], []
        for i in range(self.lookback, len(train_scaled)):
            x_train.append(train_scaled[i - self.lookback:i, 0])
            y_train.append(train_scaled[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        test_scaled = self.scaler.transform(self.test_data.reshape(-1, 1))
        x_test, y_test = [], self.test_data[self.lookback:]
        for i in range(self.lookback, len(test_scaled)):
            x_test.append(test_scaled[i - self.lookback:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        return x_train, y_train, x_test, y_test

    def build_model(self, x_train):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        self.model.add(tf.keras.layers.LSTM(self.lstm_units, return_sequences=False))
        self.model.add(tf.keras.layers.Dense(self.dense_units))
        self.model.add(tf.keras.layers.Dense(1))
        self.model.summary()

    def train_model(self, x_train, y_train):
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs)

    def predict(self, x_test):
        self.predictions = self.model.predict(x_test)
        self.predictions = self.scaler.inverse_transform(self.predictions)

    def evaluate(self, y_test):
        rmse = np.sqrt(np.mean(self.predictions - y_test) ** 2)
        return rmse

    def plot_predictions(self, data, train_len):
        validation = data[train_len:]
        validation['Predictions'] = self.predictions
        plt.figure(figsize=(16, 8))
        plt.title('Model')
        plt.xlabel('Date')
        plt.ylabel('Close Price USD ($)')
        plt.plot(data[:train_len])
        plt.plot(validation[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.show()


def split_data(x, y, test_size=0.2, shuffle=False, random_state=0):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=shuffle,
                                                        random_state=random_state)



"""
RNN
"""


class SimpleRNNModel:

    """
    Predicts stock prices using a SimpleRNN neural network.

    Parameters:
        data (pandas.DataFrame): A DataFrame containing the stock price data.
        test_size (float): The proportion of the data to use for testing. Defaults to 0.2.
        random_state (int): The random seed to use for splitting the data. Defaults to 42.
        epochs (int): The number of times to iterate over the training dataset. Defaults to 50.
        batch_size (int): The number of samples to use in each batch. Defaults to 32.
        num_units (int): The number of units in each SimpleRNN layer. Defaults to 50.
        activation (str): The activation function to use in the SimpleRNN layers. Defaults to "tanh".
        dropout (float): The proportion of input units to drop during training. Defaults to 0.2.

    Returns:
        A tuple containing the predicted stock prices and the training history.

    """
    def __init__(self, units=50, activation="tanh", dropout_rate=0.2, optimizer="adam", loss="mean_squared_error"):
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.loss = loss
        self.model = None

    def create_model(self, input_shape):
        model = Sequential()
        model.add(
            SimpleRNN(units=self.units,
                      activation=self.activation,
                      return_sequences=True,
                      input_shape=input_shape)
        )
        model.add(Dropout(self.dropout_rate))
        model.add(
            SimpleRNN(units=self.units,
                      activation=self.activation,
                      return_sequences=True)
        )
        model.add(Dropout(self.dropout_rate))
        model.add(
            SimpleRNN(units=self.units,
                      activation=self.activation,
                      return_sequences=True)
        )
        model.add(Dropout(self.dropout_rate))
        model.add(
            SimpleRNN(units=self.units,
                      activation=self.activation)
        )
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(units=1))
        model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      metrics=["accuracy"])
        self.model = model

    def fit(self, x_train, y_train, epochs=50, batch_size=32):
        history = self.model.fit(x_train, y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 verbose=1)
        return history

    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred


def split_train_test_data(x, y, test_size=0.15, shuffle=False, random_state=0):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
                                                        shuffle=shuffle, random_state=random_state)
    return x_train, x_test, y_train, y_test


def preprocess_data(data, training_data_len, scaler):
    close_prices = data['Close']
    values = close_prices.values
    scaled_data = scaler.fit_transform(values.reshape(-1, 1))
    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train


def inverse_transform(scaler, predictions):
    return scaler.inverse_transform(predictions)


def plot_results(data, training_data_len, predictions):
    validation = data[training_data_len:]
    validation['Predictions'] = predictions
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(data[:training_data_len])
    plt.plot(validation[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()


def main():
    data = msft.filter(['Close'])
    length_data = len(data)
    split_ratio


"""
Gradient Boosting regressor
"""

def get_stock_data(ticker):
    """
    Retrieves and preprocesses stock data for a given ticker.

    Parameters:
        ticker (str): The stock ticker symbol.

    Returns:
        A DataFrame containing the preprocessed stock price data.
    """
    ticker_data = yf.Ticker(ticker).history(period="max")
    ticker_data = ticker_data.reset_index()
    ticker_data["Date"] = pd.to_datetime(ticker_data["Date"])
    ticker_data.set_index("Date", inplace=True)
    ticker_data = ticker_data.drop(columns=["Dividends", "Stock Splits"])
    ticker_data = ticker_data.dropna()
    return ticker_data


def train_gbm_model(data, test_size=0.2, random_state=42):
    """
    Trains a Gradient Boosting Regressor model to predict stock prices using historical data.

    Parameters:
        data (pandas.DataFrame): A DataFrame containing the stock price data.
        test_size (float): The proportion of the data to use for testing. Defaults to 0.2.
        random_state (int): The random seed to use for splitting the data. Defaults to 42.

    Returns:
        A trained Gradient Boosting Regressor model.
    """
    X = data.drop(columns=["Close"])
    y = data["Close"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False, random_state=random_state)

    gbm = GradientBoostingRegressor(n_estimators=50, learning_rate=0.2, max_depth=3, random_state=random_state)
    gbm.fit(X_train, y_train)

    return gbm


def evaluate_gbm_model(model, X_test, y_test):
    """
    Evaluates a trained Gradient Boosting Regressor model on test data.

    Parameters:
        model: A trained Gradient Boosting Regressor model.
        X_test (pandas.DataFrame): The test features.
        y_test (pandas.DataFrame): The true test labels.

    Returns:
        The mean squared error of the model on the test data.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    return mse


def plot_feature_importances(model, feature_names):
    """
    Plots the feature importances of a trained Gradient Boosting Regressor model.

    Parameters:
        model: A trained Gradient Boosting Regressor model.
        feature_names (list): A list of feature names.
    """
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.show()


# Example usage:
if __name__ == "__main__":
    msft_data = get_stock_data("MSFT")
    msft_model = train_gbm_model(msft_data)
    msft_mse = evaluate_gbm_model(msft_model, X_test, y_test)
    plot_feature_importances(msft_model, msft_data.columns)



# ARIMA

logging.basicConfig(level=logging.INFO)

def train_arima_model(df):
    """
    Trains an ARIMA model on a given dataset, and returns the predicted values and mean squared error.

    Parameters:
        df (pandas.DataFrame): The input dataset to train the ARIMA model on.

    Returns:
        tuple: A tuple containing the predicted values and mean squared error.
    """

    # Split the data into train and test sets
    train_data, test_data = df[0:int(len(df)*0.7)], df[int(len(df)*0.7):]

    # Extract the 'Close' column from the train and test sets
    training_data = train_data['Close'].values
    test_data = test_data['Close'].values

    # Create a list of historical data
    history = [x for x in training_data]

    # Create an empty list to store the predicted values
    model_predictions = []

    # Loop over the test data and make predictions
    N_test_observations = len(test_data)
    for time_point in range(N_test_observations):
        # Fit an ARIMA model to the historical data
        try:
            model = sm.tsa.arima.ARIMA(history, order=(4,1,0))
            model_fit = model.fit()
        except Exception as e:
            logging.error(f"ARIMA model failed to fit at time point {time_point}: {e}")
            return None, None

        # Make a prediction for the next time point
        try:
            output = model_fit.forecast()
            yhat = output[0]
        except Exception as e:
            logging.error(f"ARIMA model failed to predict at time point {time_point}: {e}")
            return None, None

        # Append the predicted value to the list of predictions
        model_predictions.append(yhat)

        # Append the true test value to the historical data
        true_test_value = test_data[time_point]
        history.append(true_test_value)

    # Calculate the mean squared error of the model predictions
    MSE_error = mean_squared_error(test_data, model_predictions)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(model_predictions, label='Predicted')
    ax.plot(test_data, label='True')
    ax.legend()
    ax.set_title('Stock Price Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    plt.show()

    # Return the predicted values and the mean squared error
    return model_predictions, mse_error


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load the data
    data = pd.read_csv('msft.csv')

    # Predict the stock prices
    predictions, mse_error = predict_stock_prices(data)

    # Log the mean squared error
    logging.info(f'Testing Mean Squared Error: {mse_error}')





