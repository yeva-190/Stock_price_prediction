import unittest
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from predict_stock_prices import predict_stock_prices
import joblib

# Linear Regression
df = pd.read_csv('msft.csv')

# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

class TestPredictStockPrices(unittest.TestCase):
    def test_returns_tuple(self):
        result = predict_stock_prices(df)
        self.assertIsInstance(result, tuple)

    def test_returns_correct_tuple_size(self):
        result = predict_stock_prices(df)
        self.assertEqual(len(result), 4)

    def test_coef_is_float(self):
        result = predict_stock_prices(df)
        self.assertIsInstance(result[1], float)

    def test_intercept_is_float(self):
        result = predict_stock_prices(df)
        self.assertIsInstance(result[2], float)

    def test_confidence_is_float(self):
        result = predict_stock_prices(df)
        self.assertIsInstance(result[3], float)

    def test_predicted_is_numpy_array(self):
        result = predict_stock_prices(df)
        self.assertIsInstance(result[0], np.ndarray)

    def test_predicted_array_length_is_correct(self):
        result = predict_stock_prices(df)
        self.assertEqual(len(result[0]), len(df) * 0.2)

    def test_plot_is_shown(self):
        with patch('matplotlib.pyplot.show') as show_mock:
            result = predict_stock_prices(df)
            show_mock.assert_called_once()

@pytest.mark.integration
def test_integration_predict_stock_prices():
    result = predict_stock_prices(df)
    predicted, coef, intercept, confidence = result

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    expected_coef = lr.coef_
    expected_intercept = lr.intercept_
    expected_confidence = lr.score(X_test, y_test)
    expected_predicted = lr.predict(X_test)

    np.testing.assert_allclose(coef, expected_coef)
    np.testing.assert_allclose(intercept, expected_intercept)
    np.testing.assert_allclose(confidence, expected_confidence)
    np.testing.assert_allclose(predicted, expected_predicted, rtol=1e-3, atol=1e-3)


# LSTM

class TestLSTMPredictor(TestCase):

    def setUp(self):
        self.data = pd.read_csv("msft.csv")
        self.train_data = self.data.loc[:400, "Close"].values
        self.test_data = self.data.loc[400:, "Close"].values
        self.predictor = LSTMPredictor(self.train_data, self.test_data)

    def test_prepare_data(self):
        x_train, y_train, x_test, y_test = self.predictor.prepare_data()
        self.assertEqual(x_train.shape,
                         (self.train_data.shape[0] - self.predictor.lookback, self.predictor.lookback, 1))
        self.assertEqual(y_train.shape, (self.train_data.shape[0] - self.predictor.lookback,))
        self.assertEqual(x_test.shape, (self.test_data.shape[0] - self.predictor.lookback, self.predictor.lookback, 1))
        self.assertEqual(y_test.shape, (self.test_data.shape[0] - self.predictor.lookback,))

    def test_build_model(self):
        x_train, _, _, _ = self.predictor.prepare_data()
        self.predictor.build_model(x_train)
        self.assertIsInstance(self.predictor.model, tf.keras.Sequential)
        self.assertEqual(len(self.predictor.model.layers), 4)

    def test_train_model(self):
        x_train, y_train, _, _ = self.predictor.prepare_data()
        with patch.object(self.predictor.model, 'fit', return_value=None) as mock_fit:
            self.predictor.train_model(x_train, y_train)
            mock_fit.assert_called_once_with(x_train, y_train, batch_size=self.predictor.batch_size,
                                             epochs=self.predictor.epochs)

    def test_predict(self):
        x_train, _, x_test, _ = self.predictor.prepare_data()
        self.predictor.build_model(x_train)
        self.predictor.train_model(x_train, self.train_data[self.predictor.lookback:])
        self.predictor.predict(x_test)
        self.assertEqual(self.predictor.predictions.shape, (self.test_data.shape[0] - self.predictor.lookback, 1))

    def test_evaluate(self):
        y_test = self.test_data[self.predictor.lookback:]
        rmse = self.predictor.evaluate(y_test)
        self.assertIsInstance(rmse, float)

    def test_plot_predictions(self):
        self.predictor.lookback = 10
        x_train, _, x_test, _ = self.predictor.prepare_data()
        self.predictor.build_model(x_train)
        self.predictor.train_model(x_train, self.train_data[self.predictor.lookback:])
        self.predictor.predict(x_test)
        with patch("matplotlib.pyplot.show") as mock_show:
            self.predictor.plot_predictions(self.data, self.train_data.shape[0])
            mock_show.assert_called_once()


@pytest.mark.integration
def test_lstm_predictor():
    data = pd.read_csv("msft.csv")
    train_data = data.loc[:400, "Close"].values
    test_data = data.loc[400:, "Close"].values
    predictor = LSTMPredictor(train_data, test_data)
    x_train, y_train, x_test, y_test = predictor.prepare_data()
    predictor.build_model(x_train



#RNN

    class TestSimpleRNNModel(unittest.TestCase):
        def setUp(self):
            self.data = pd.read_csv('msft.csv')
            self.model = SimpleRNNModel()

        def test_create_model(self):
            input_shape = (60, 1)
            self.model.create_model(input_shape)
            self.assertIsInstance(self.model.model, Sequential)
            self.assertEqual(len(self.model.model.layers), 7)

        def test_fit(self):
            scaler = MinMaxScaler()
            x_train, y_train = preprocess_data(self.data, 100, scaler)
            self.model.create_model((x_train.shape[1], 1))
            history = self.model.fit(x_train, y_train, epochs=2, batch_size=32)
            self.assertIsInstance(history, type(self.model.model.history))

        def test_predict(self):
            scaler = MinMaxScaler()
            x_train, y_train = preprocess_data(self.data, 100, scaler)
            self.model.create_model((x_train.shape[1], 1))
            self.model.fit(x_train, y_train, epochs=2, batch_size=32)
            x_test = np.zeros((10, 60, 1))
            y_pred = self.model.predict(x_test)
            self.assertEqual(y_pred.shape, (10, 1))

        def test_split_train_test_data(self):
            x = np.random.rand(100, 10)
            y = np.random.rand(100, 1)
            x_train, x_test, y_train, y_test = split_train_test_data(x, y, test_size=0.2)
            self.assertEqual(x_train.shape[0], 80)
            self.assertEqual(x_test.shape[0], 20)
            self.assertEqual(y_train.shape[0], 80)
            self.assertEqual(y_test.shape[0], 20)

        def test_preprocess_data(self):
            scaler = MinMaxScaler()
            x_train, y_train = preprocess_data(self.data, 100, scaler)
            self.assertEqual(x_train.shape[0], y_train.shape[0])
            self.assertEqual(x_train.shape[1], 60)
            self.assertEqual(x_train.shape[2], 1)

        def test_inverse_transform(self):
            scaler = MinMaxScaler()
            x_train, y_train = preprocess_data(self.data, 100, scaler)
            self.model.create_model((x_train.shape[1], 1))
            self.model.fit(x_train, y_train, epochs=2, batch_size=32)
            x_test = np.zeros((10, 60, 1))
            y_pred = self.model.predict(x_test)
            y_inv = inverse_transform(scaler, y_pred)
            self.assertEqual(y_inv.shape, (10, 1))

    if __name__ == '__main__':
        unittest.main()


@pytest.mark.integration
def test_simple_rnn_model_integration(self):
    data = pd.read_csv('msft.csv')
    scaler = MinMaxScaler()
    x_train, y_train = preprocess_data(data, 100, scaler)
    model = SimpleRNNModel(units=50, activation='tanh', dropout_rate=0.2, optimizer='adam', loss='mean_squared_error')


# Gradient Boosting

class TestStockPredictor(unittest.TestCase):

    def test_get_stock_data(self):
        # test retrieval of data for MSFT ticker
        data = get_stock_data("MSFT")
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)

    def test_train_gbm_model(self):
        # test training of model on MSFT data
        data = get_stock_data("MSFT")
        X = data.drop(columns=["Close"])
        y = data["Close"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
        model = train_gbm_model(data=X_train, test_size=0.2, random_state=42)
        self.assertIsInstance(model, GradientBoostingRegressor)
        self.assertTrue(model.n_estimators > 0)
        self.assertTrue(model.learning_rate > 0)
        self.assertTrue(model.max_depth > 0)

    def test_evaluate_gbm_model(self):
        # test evaluation of model on MSFT test data
        data = get_stock_data("MSFT")
        X = data.drop(columns=["Close"])
        y = data["Close"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
        model = train_gbm_model(data=X_train, test_size=0.2, random_state=42)
        mse = evaluate_gbm_model(model, X_test, y_test)
        self.assertTrue(mse >= 0)


@pytest.mark.integration
class TestGBMIntegration(unittest.TestCase):

    def setUp(self):
        self.msft_data = pd.read_csv("msft.csv")
        self.msft_model = joblib.load("msft_model.pkl")

    def test_evaluate_gbm_model(self):
        X_test = self.msft_data.drop(columns=["Close"])
        y_test = self.msft_data["Close"]
        mse = evaluate_gbm_model(self.msft_model, X_test, y_test)
        self.assertFalse(pd.isna(mse))
        self.assertGreater(mse, 0)

    def test_train_gbm_model(self):
        msft_model = train_gbm_model(self.msft_data)
        self.assertIsNotNone(msft_model)
        self.assertTrue(any(msft_model.feature_importances_ > 0))


if __name__ == "__main__":
    unittest.main()