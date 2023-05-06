import pandas as pd
import numpy as np
import random
import time
from flask import Flask, jsonify, request
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
from lstm_predictor import LSTMPredictor
from prometheus_flask_exporter import PrometheusMetrics
from sklearn.ensemble import GradientBoostingRegressor

app = Flask(__name__)
metrics = PrometheusMetrics(app)
data = pd.read_csv('msft.csv')

# Endpoint for the linear regression model
@app.route('/linear-regression', methods=['POST'])
def linear_regression():

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




# Add Prometheus metrics middleware to the app
metrics = PrometheusMetrics(app)

# Endpoint for the LSTM model
@app.route('/lstm', methods=['POST'])
def lstm():

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

def create_rnn_model():
    model = Sequential()
    model.add(SimpleRNN(units=50, activation='relu', return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(units=50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# Endpoint for the RNN model
@app.route('/rnn', methods=['POST'])
def rnn():

    # Prepare the data
    train_set = data.iloc[:500, 4:5].values
    test_set = data.iloc[500:, 4:5].values

    # Normalize the data
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0,1))
    train_set_scaled = sc.fit_transform(train_set)
    test_set_scaled = sc.transform(test_set)

    # Prepare the data for RNN model
    X_train = []
    y_train = []
    for i in range(60, 500):
        X_train.append(train_set_scaled[i-60:i, 0])
        y_train.append(train_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Train the RNN model
    model = create_rnn_model()
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Prepare the data for predictions
    inputs = data.iloc[:, 4:5].values
    inputs = inputs[len(inputs) - len(test_set) - 60:]
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(60, len(test_set) + 60):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predict the stock prices
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    # Prepare the response data
    response_data = {
        'predictions': predicted_stock_price.tolist()
    }

    # Return the response as JSON
    return jsonify(response_data)

train_data = data.iloc[:-30, :]
train_X = train_data[['Open', 'High', 'Low']].values
train_y = train_data['Close'].values

# Train the model
model = GradientBoostingRegressor(n_estimators=100, max_depth=5)
model.fit(train_X, train_y)

# Define the endpoint for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data
    data = pd.read_json(request.data)

    # Prepare the input data
    X = data[['Open', 'High', 'Low']].values

    # Make the prediction
    predicted = model.predict(X)

    # Prepare the response data
    response_data = {
        'predicted': predicted.tolist(),
    }

    # Return the response as JSON
    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True)