from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pytest
import unittest
from my_code import prepare_data, build_model


# Linear Regression
def test_train_test_split():
    # create some test data
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([3, 7, 11, 15])

    # perform train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # check that the shapes of the train and test data are correct
    assert x_train.shape == (3, 2)
    assert x_test.shape == (1, 2)
    assert y_train.shape == (3,)
    assert y_test.shape == (1,)


def test_linear_regression():
    # create some test data
    x_train = np.array([1, 2, 3, 4])
    y_train = np.array([2, 4, 6, 8])
    x_test = np.array([5, 6, 7, 8])
    y_test = np.array([10, 12, 14, 16])

    # train a linear regression model
    lr = LinearRegression()
    lr.fit(x_train.reshape(-1, 1), y_train)

    # check that the regression coefficient and intercept are correct
    assert lr.coef_ == 2.0
    assert lr.intercept_ == 0.0

    # calculate the regression confidence
    regression_confidence = lr.score(x_test.reshape(-1, 1), y_test)

    # check that the confidence is greater than 0.5
    assert regression_confidence > 0.5


def test_predictions():
    # create some test data
    x_train = np.array([1, 2, 3, 4])
    y_train = np.array([2, 4, 6, 8])
    x_test = np.array([5, 6, 7, 8])
    y_test = np.array([10, 12, 14, 16])

    # train a linear regression model
    lr = LinearRegression()
    lr.fit(x_train.reshape(-1, 1), y_train)

    # make predictions on the test data
    predicted = lr.predict(x_test.reshape(-1, 1))

    # check that the predictions are close to the true values
    assert np.allclose(predicted, y_test, rtol=1e-2)


@pytest.mark.mpl_image_compare
def test_plot():
    # create some test data
    x_train = np.array([1, 2, 3, 4])
    y_train = np.array([2, 4, 6, 8])
    x_test = np.array([5, 6, 7, 8])
    y_test = np.array([10, 12, 14, 16])

    # train a linear regression model
    lr = LinearRegression()
    lr.fit(x_train.reshape(-1, 1), y_train)

    # make predictions on the test data
    predicted = lr.predict(x_test.reshape(-1, 1))

    # plot the predictions
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.scatter(x_train, y_train



# LSTM


class TestMyCode(unittest.TestCase):

    def setUp(self):
        # Define some common variables
        self.x, self.y = some_data  # Replace with actual data
        self.test_size = 0.15
        self.shuffle = False
        self.random_state = 0
        self.model_epochs = 3

    def test_prepare_data(self):
        # Test that the prepare_data function returns the correct shapes
        x_train, x_test, y_train, y_test = prepare_data(self.x, self.y, self.test_size, self.shuffle, self.random_state)
        self.assertEqual(x_train.shape[0], y_train.shape[0])
        self.assertEqual(x_test.shape[0], y_test.shape[0])

    def test_build_model(self):
        # Test that the build_model function returns a keras model object
        model = build_model()
        self.assertIsInstance(model, keras.Model)

    def test_train_and_predict(self):
        # Test that the model can be trained and used to make predictions
        x_train, x_test, y_train, y_test = prepare_data(self.x, self.y, self.test_size, self.shuffle, self.random_state)
        model = build_model()
        model.fit(x_train, y_train, batch_size=1, epochs=self.model_epochs)
        predictions = model.predict(x_test)
        self.assertEqual(predictions.shape[0], y_test.shape[0])


#RNN

import pytest
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout

# Dummy data for testing
x_train = np.random.rand(100, 10, 1)
y_train = np.random.rand(100, 1)

# Unit tests
def test_model_add():
    model = Sequential()
    model.add(
        SimpleRNN(
            units=50,
            activation="tanh",
            return_sequences=True,
            input_shape=(x_train.shape[1], 1),
        )
    )
    assert len(model.layers) == 1

def test_model_compile():
    model = Sequential()
    model.add(
        SimpleRNN(
            units=50,
            activation="tanh",
            return_sequences=True,
            input_shape=(x_train.shape[1], 1),
        )
    )
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    assert model.optimizer.__class__.__name__ == "Adam"
    assert model.loss == "mean_squared_error"
    assert model.metrics[0] == "accuracy"

# Integration test
def test_model_fit():
    model = Sequential()
    model.add(
        SimpleRNN(
            units=50,
            activation="tanh",
            return_sequences=True,
            input_shape=(x_train.shape[1], 1),
        )
    )
    model.add(Dropout(0.2))
    model.add(
        SimpleRNN(units=50, activation="tanh", return_sequences=True)
    )
    model.add(Dropout(0.2))
    model.add(
        SimpleRNN(units=50, activation="tanh", return_sequences=True)
    )
    model.add(Dropout(0.2))
    model.add(SimpleRNN(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=3, batch_size=32)
    assert len(history.history["loss"]) == 3
    assert history.history["loss"][0] != history.history["loss"][-1]
    assert len(history.history["accuracy"]) == 3
    assert history.history["accuracy"][0] != history.history["accuracy"][-1]


#Gradient Boosting
import pytest
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


@pytest.fixture
def data():
    ticker = yf.Ticker("MSFT")
    data = ticker.history(period="max")
    data = data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index("Date", inplace=True)
    data = data.drop(columns=["Dividends", "Stock Splits"])
    data = data.dropna()
    return data


@pytest.fixture
def split_data(data):
    X = data.drop(columns=["Close"])
    y = data["Close"]
    return train_test_split(X, y, test_size=0.2, shuffle=False)


def test_data_is_not_empty(data):
    assert not data.empty, "Data is empty"


def test_split_data_has_correct_shape(split_data):
    X_train, X_test, y_train, y_test = split_data
    assert len(X_train) == len(y_train), "X_train and y_train have different lengths"
    assert len(X_test) == len(y_test), "X_test and y_test have different lengths"


def test_gradient_boosting_regressor_fits_and_predicts_correctly(split_data):
    X_train, X_test, y_train, y_test = split_data
    gbm = GradientBoostingRegressor(n_estimators=50, learning_rate=0.2, max_depth=3, random_state=42)
    gbm.fit(X_train, y_train)
    y_pred = gbm.predict(X_test)
    assert len(y_pred) == len(y_test), "Lengths of predicted and true values are different"
    assert mean_squared_error(y_test, y_pred) >= 0, "Mean Squared Error should be non-negative"


def test_xgboost_feature_importances_plot_has_correct_title(data):
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    xgb_model.fit(data.drop(columns=["Close"]), data["Close"])
    importances = xgb_model.feature_importances_
    feature_names = data.columns
    indices = importances.argsort()[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(data.shape[1]), importances[indices])
    plt.xticks(range(data.shape[1]), feature_names[indices], rotation=90)
    assert plt.gca().get_title() == "Feature Importances", "Incorrect title for feature importances plot"
