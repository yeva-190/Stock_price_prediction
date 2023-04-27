## Flask App for Stock Price Prediction
This is a Flask app for predicting stock prices using various machine learning models. The app has endpoints for the following models:

1. Linear Regression
2. Long Short-Term Memory (LSTM) Neural Network
3. Simple Recurrent Neural Network (RNN)
4. Gradient Boosting Regression
5. ARIMA 

### Prerequisites
Before running the app, make sure you have the following installed:

- Python 3.x
- Flask
- Tensorflow
- Keras
- Pandas
- Scikit-learn
- Numpy
### Usage
1. clone this repository to your local machine.
2. Install the required packages using pip: pip install -r requirements.txt
3. Run the Flask app: python app.py
4. Send a POST request to the desired endpoint with a JSON payload containing the stock data.

### Linear Regression Endpoint
Endpoint: /linear-regression

This endpoint uses a linear regression model to predict future stock prices.

Request

The request should have the following JSON payload:

```json
{
  "Open": [...],
  "High": [...],
  "Low": [...],
  "Close": [...],
  "Volume": [...]
}
```

Response

The response will contain the following JSON payload:


```json
{
    "predicted": [...],
    "coef": [...],
    "intercept": [...],
    "confidence": [...]
}
```
- predicted: An array containing the predicted stock prices.
- coef: An array containing the coefficients of the linear regression model.
- intercept: An array containing the intercept of the linear regression model.
- confidence: An array containing the confidence interval of the predictions.

### LSTM Endpoint
Endpoint: /lstm

This endpoint uses a Long Short-Term Memory (LSTM) neural network to predict future stock prices.

Request

The request should have the following JSON payload:

```json
{
    "Open": [...],
    "High": [...],
    "Low": [...],
    "Close": [...],
    "Volume": [...],
}
```

Response

The response will contain the following JSON payload:

```json
{
    "predictions": [...],
    "rmse": ...
}
```
- predictions: An array containing the predicted stock prices.
- rmse: The root mean squared error of the predictions.

### Simple RNN Endpoint
Endpoint: /RNN

This endpoint uses a simple recurrent neural network (RNN) to predict future stock prices.

Request

The request should have the following JSON payload:
```json
{
    "Open": [...],
    "High": [...],
    "Low": [...],
    "Close": [...],
    "Volume": [...],
}
```

Response

The response will contain the following JSON payload:

```json
{
    "prediction": [...]
}
```
prediction: An array containing the predicted stock prices.

### Gradient Boosting

Endpoint: /gbm

This endpoint uses a Gradient Boosting Regressor model to predict future stock prices.

Request

The request should have the following JSON payload:

```json
{
"Open": [...],
"High": [...],
"Low": [...],
"Close": [...],
"Volume": [...]
}
```

Response

The response will contain the following JSON payload:

```json
{
"predicted_price": [...]
}
```
predicted_price: An array containing the predicted stock prices.

### ARIMA

Endpoint: /train_arima_model

This endpoint trains an Autoregressive Integrated Moving Average (ARIMA) model to predict future stock prices.

Request

The request should have the following JSON payload:

```json
{
"Close": [...]
}
```

Response

The response will contain the following JSON payload:

```json
{
"predictions": [...],
"mse_error": ...
}
```

- predictions: An array containing the predicted stock prices.
- mse_error: The mean squared error of the predictions.