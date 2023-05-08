from flask import Flask, request, jsonify
import pandas as pd
#from lstm import LSTMPredictor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from linear_regression import predict_stock_prices
#from tensorflow import tf
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib as joblib
import logging
from flask import Flask, jsonify




app = Flask(__name__)

# Endpoint for the linear regression model
@app.route('/linear-regression', methods=['POST'])
def linear_regression():
    # Get the request data
    data = yf.download('MSFT', start='2021-01-01', end='2023-05-08')
    # convert the data to a JSON string
    data_json = data.to_json()
    # return the JSON response
    return jsonify(data_json)

    # Call the model function and get the results
    predicted, coef, intercept, confidence = predict_stock_prices(data)

    # Prepare the response data
    response_data = {
        'predicted': predicted.tolist(),
        'coef': coef.tolist(),
        'intercept': intercept.tolist(),
        'confidence': confidence.tolist()
    }

    # Return the response as JSON
    return jsonify(response_data)

# Endpoint for the LSTM model
@app.route('/lstm', methods=['POST'])
def lstm():
    # Get the request data
    data = yf.download('MSFT', start='2021-01-01', end='2023-05-08')
    # convert the data to a JSON string
    data_json = data.to_json()
    # return the JSON response
    return jsonify(data_json)

    # Call the model function and get the results
    predictor = LSTMPredictor(data['Close'].values, lookback=60, lstm_units=100, dense_units=25, epochs=3, batch_size=1)
    x_train, y_train, x_test, y_test = predictor.prepare_data()
    predictor.build_model(x_train)
    predictor.train_model(x_train, y_train)
    predictor.predict(x_test)
    rmse = predictor.evaluate(y_test)

    # Prepare the response data
    response_data = {
        'predictions': predictor.predictions.tolist(),
        'rmse': rmse
    }

    # Return the response as JSON
    return jsonify(response_data)


class SimpleRNNModel:
    # class definition here

    data = yf.download('MSFT', start='2021-01-01', end='2023-05-08')
    # convert the data to a JSON string
    data_json = data.to_json()
    # return the JSON response
    return jsonify(data_json)

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


@app.route('/RNN', methods=['POST'])
def predict_stock_price():
    # Load the saved model and scaler
    model = load_model('simple_rnn_model.h5')
    scaler = MinMaxScaler()
    scaler.min_, scaler.scale_ = np.load('scaler_params.npy')

    # Get the stock data from the request
    data = yf.download('MSFT', start='2021-01-01', end='2023-05-08')
    # convert the data to a JSON string
    data_json = data.to_json()
    # return the JSON response
    return jsonify(data_json)

    # Preprocess the data
    x_pred = preprocess_data(data, 60, scaler)

    # Make the prediction
    y_pred = model.predict(x_pred)
    y_pred = scaler.inverse_transform(y_pred)

    # Return the prediction
    return jsonify({'prediction': y_pred.tolist()})


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


@app.route('/gbm', methods=['POST'])
def predict():
    # Load the trained model
    model = joblib.load('gbm_model.joblib')

    # Get the request data
    data = yf.download('MSFT', start='2021-01-01', end='2023-05-08')
    # convert the data to a JSON string
    data_json = data.to_json()
    # return the JSON response
    return jsonify(data_json)

    # Parse the request data into a Pandas DataFrame
    X = pd.DataFrame.from_dict(data)

    # Make the prediction
    y_pred = model.predict(X)

    # Return the prediction as a JSON response
    return jsonify({'predicted_price': list(y_pred)})


if __name__ == '__main__':
    # Load and train the model
    msft_data = get_stock_data("MSFT")
    msft_model = train_gbm_model(msft_data)
    joblib.dump(msft_model, 'gbm_model.joblib')



logging.basicConfig(level=logging.INFO)

@app.route('/train_arima_model', methods=['POST'])
def train_arima_model_endpoint():
    # Get the input data from the request
    data = yf.download('MSFT', start='2021-01-01', end='2023-05-08')
    # convert the data to a JSON string
    data_json = data.to_json()
    # return the JSON response
    return jsonify(data_json)

    # Convert the input data to a pandas DataFrame

    # Train the ARIMA model
    predictions, mse_error = train_arima_model(data)

    # If the model failed to fit or predict, return an error response
    if predictions is None or mse_error is None:
        return jsonify({'error': 'Failed to train the ARIMA model.'}), 400

    # Convert the predictions to a list
    predictions_list = [float(prediction) for prediction in predictions]

    # Return the predicted values and the mean squared error as a JSON response
    response = {
        'predictions': predictions_list,
        'mse_error': float(mse_error)
    }
    return jsonify(response)




if __name__ == '__main__':
    app.run(debug=True)



